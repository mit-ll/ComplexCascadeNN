function [x] = read_interleaved_float32(filename,offset_bof_samps,nread_samps)

fid = fopen(filename,'rb');

if fid<3 
    error('Cannot open file');
end

fseek(fid,offset_bof_samps*2*4,'bof');

data = fread(fid,[2,nread_samps],'float32=>double');

x = complex(data(1,:),data(2,:));

fclose(fid);